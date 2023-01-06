import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ExtractorPageComponent } from './extractor-page/extractor-page.component';
import { TesterPageComponent } from './tester-page/tester-page.component';

const routes: Routes = [
  {
    path: '',
    component: ExtractorPageComponent,
  },
  {
    path: 'tester',
    component: TesterPageComponent,
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
